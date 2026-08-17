"""Planner-specific result plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def save_rrt_results(planner, path, smooth_path, filename, show=True):
    """Save an RRT result while preserving its existing figure style."""
    plt.figure(figsize=(12, 6))

    ax = plt.gca()
    planner.env.plot_environment(ax)
    planner._plot_safety_boundaries(ax)

    for node in planner.node_list:
        if node.parent:
            plt.plot(node.path_x, node.path_y, "-g", alpha=0.3)

    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, "b--", linewidth=2, label="原始路径")

    if smooth_path:
        smooth_path_x = [p[0] for p in smooth_path]
        smooth_path_y = [p[1] for p in smooth_path]
        plt.plot(smooth_path_x, smooth_path_y, "r-", linewidth=2, label="平滑路径")

    if path:
        plt.plot(path[0][0], path[0][1], "go", markersize=10, label="起点")
        plt.plot(path[-1][0], path[-1][1], "ro", markersize=10, label="终点")

    plt.legend()
    plt.title(f"RRT路径规划结果 (安全距离: {planner.safety_distance:.1f}m)")
    plt.axis("equal")
    plt.grid(True)

    plt.savefig(filename, dpi=100, bbox_inches="tight")
    print(f"路径规划结果已保存为 {filename}")

    # RRT historically closes instead of displaying; preserve that behavior
    # even though ``show`` defaults to True for the shared public API.
    plt.close()


def save_astar_results(planner, path, smooth_path, filename, show=True):
    """Save an A* result while preserving its existing dual-panel style."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.set_title("A*路径规划 - 栅格地图", fontsize=14, fontweight="bold")
    ax1.imshow(
        planner.grid_map,
        cmap="binary",
        origin="lower",
        extent=[planner.x_min, planner.x_max, planner.y_min, planner.y_max],
    )

    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax1.plot(path_x, path_y, "r-", linewidth=2, label="A*原始路径")

    start_point = planner.env.start_point
    end_point = planner.env.end_point
    ax1.plot(start_point[0], start_point[1], "go", markersize=10, label="起点")
    ax1.plot(end_point[0], end_point[1], "ro", markersize=10, label="终点")

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    ax2.set_title("A*路径规划 - 车道约束路径", fontsize=14, fontweight="bold")
    planner.env.plot_environment(ax2)

    for i, center in enumerate(planner.lane_centers):
        ax2.axhline(
            y=center,
            color="yellow",
            linestyle="-",
            linewidth=2,
            alpha=0.8,
            label=f"车道{i+1}中心" if i == 0 else "",
        )
        ax2.axhline(
            y=center + planner.max_lane_deviation,
            color="orange",
            linestyle=":",
            alpha=0.5,
        )
        ax2.axhline(
            y=center - planner.max_lane_deviation,
            color="orange",
            linestyle=":",
            alpha=0.5,
        )
        ax2.fill_between(
            [0, planner.env.road_length],
            center - planner.max_lane_deviation,
            center + planner.max_lane_deviation,
            alpha=0.1,
            color="green",
            label="允许范围" if i == 0 else "",
        )

    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax2.plot(
            path_x,
            path_y,
            "--",
            color="red",
            linewidth=1.5,
            label="A*原始路径",
            alpha=0.7,
        )

    if smooth_path:
        smooth_x = [p[0] for p in smooth_path]
        smooth_y = [p[1] for p in smooth_path]
        ax2.plot(
            smooth_x,
            smooth_y,
            "-",
            color="blue",
            linewidth=3,
            label="A*车道约束路径",
        )

    param_text = (
        f"车道偏离限制: {planner.max_lane_deviation:.2f}m\n"
        f"变道角度限制: {np.rad2deg(planner.lane_change_angle_limit):.1f}°"
    )
    ax2.text(
        0.02,
        0.98,
        param_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"A*路径规划结果已保存为 {filename}")

    # A* historically closes instead of displaying; preserve that behavior
    # even though ``show`` defaults to True for the shared public API.
    plt.close()


def save_rrt_star_results(planner, path, filename, show=True):
    """Save an RRT* result while preserving its custom road drawing."""
    print("保存路径规划结果...")

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    road_rect = Rectangle(
        (0, 0),
        planner.env.road_length,
        planner.env.road_width,
        linewidth=2,
        edgecolor="black",
        facecolor="darkgray",
        alpha=0.5,
        zorder=0,
    )
    ax.add_patch(road_rect)

    lane_colors = ["#f0f0f0", "#e8e8e8", "#f0f0f0"]
    for i in range(planner.env.num_lanes):
        lane_y_start = i * planner.env.lane_width
        lane_rect = Rectangle(
            (0, lane_y_start),
            planner.env.road_length,
            planner.env.lane_width,
            facecolor=lane_colors[i % len(lane_colors)],
            alpha=0.3,
            zorder=1,
        )
        ax.add_patch(lane_rect)

    for i in range(1, planner.env.num_lanes):
        lane_y = i * planner.env.lane_width
        ax.plot(
            [0, planner.env.road_length],
            [lane_y, lane_y],
            "y--",
            linewidth=2,
            alpha=0.8,
            zorder=2,
        )

    ax.plot(
        [0, planner.env.road_length],
        [0, 0],
        "y-",
        linewidth=3,
        alpha=0.9,
        zorder=2,
    )
    ax.plot(
        [0, planner.env.road_length],
        [planner.env.road_width, planner.env.road_width],
        "y-",
        linewidth=3,
        alpha=0.9,
        zorder=2,
    )

    lower_safe_boundary = planner.safety_distance
    upper_safe_boundary = planner.env.road_width - planner.safety_distance
    ax.plot(
        [0, planner.env.road_length],
        [lower_safe_boundary, lower_safe_boundary],
        "r--",
        linewidth=2.5,
        alpha=0.8,
        label="安全距离",
        zorder=3,
    )
    ax.plot(
        [0, planner.env.road_length],
        [upper_safe_boundary, upper_safe_boundary],
        "r--",
        linewidth=2.5,
        alpha=0.8,
        zorder=3,
    )

    for i in range(planner.env.num_lanes):
        lane_center = (i + 0.5) * planner.env.lane_width
        ax.text(
            planner.env.road_length + 2,
            lane_center,
            f"车道 {i+1}",
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                alpha=0.8,
                boxstyle="round,pad=0.3",
            ),
            zorder=10,
        )

    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5, zorder=1)

    if hasattr(planner.env, "obstacle_vehicles"):
        for vehicle in planner.env.obstacle_vehicles:
            safety_rect = Rectangle(
                (
                    vehicle.x - vehicle.length / 2 - planner.safety_distance,
                    vehicle.y - vehicle.width / 2 - planner.safety_distance,
                ),
                vehicle.length + 2 * planner.safety_distance,
                vehicle.width + 2 * planner.safety_distance,
                fill=False,
                edgecolor="r",
                linestyle="--",
                linewidth=2.0,
                alpha=0.6,
                zorder=4,
            )
            ax.add_patch(safety_rect)
            vehicle.draw(ax)

    for node in planner.node_list:
        if node.parent is not None:
            ax.plot(
                [node.x, node.parent.x],
                [node.y, node.parent.y],
                "g-",
                alpha=0.15,
                linewidth=0.3,
                zorder=5,
            )

    if path:
        path_array = np.array(path)
        ax.plot(
            path_array[:, 0],
            path_array[:, 1],
            "r-",
            linewidth=4,
            label="规划路径",
            zorder=8,
        )

        ax.plot(
            path[0][0],
            path[0][1],
            "go",
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=2,
            label="起点",
            zorder=9,
        )
        ax.plot(
            path[-1][0],
            path[-1][1],
            "ro",
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=2,
            label="终点",
            zorder=9,
        )

        ax.annotate(
            "起点",
            (path[0][0], path[0][1]),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="green",
                alpha=0.9,
                boxstyle="round,pad=0.3",
            ),
            zorder=10,
        )
        ax.annotate(
            "终点",
            (path[-1][0], path[-1][1]),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="red",
                alpha=0.9,
                boxstyle="round,pad=0.3",
            ),
            zorder=10,
        )

    ax.set_xlim(-2, planner.env.road_length + 8)
    ax.set_ylim(-2, planner.env.road_width + 2)
    ax.set_xlabel("X (m)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Y (m)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"RRT* 路径规划结果（安全距离: {planner.safety_distance:.1f}m）",
        fontsize=16,
        fontweight="bold",
    )

    legend = ax.legend(
        loc="upper right",
        fontsize=12,
        framealpha=0.9,
        edgecolor="black",
        fancybox=True,
        shadow=True,
    )
    legend.set_zorder(11)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    if show:
        plt.show()
    else:
        plt.close()

    print(f"RRT*路径规划结果已保存为 {filename}")
