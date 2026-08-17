"""Static result plotting for simulation sessions."""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from autodrive import i18n


def plot_simulation_result(
    env,
    path,
    smooth_path,
    x_history,
    y_history,
    v_history,
    t_history,
    target_v_history=None,
    safety_distance=1.5,
    *,
    show=True,
    filename="simulation_results.png",
):
    """Draw and save the legacy simulation result layout."""
    labels = i18n.labels or i18n.use_english_labels()
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(
        2, 2, width_ratios=[2, 1], height_ratios=[1, 1]
    )

    ax1 = plt.subplot(gs[:, 0])
    env.plot_environment(ax1)
    title = labels.get("路径规划与跟踪", "Path Planning & Tracking")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.text(
        5,
        env.road_width - 0.5,
        f"安全距离: {safety_distance:.2f}米",
        fontsize=12,
        color="black",
        bbox=dict(
            facecolor="white", alpha=0.7, boxstyle="round,pad=0.5"
        ),
        zorder=15,
    )

    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax1.plot(
        path_x,
        path_y,
        "--",
        color="navy",
        linewidth=1.5,
        label=labels.get("原始路径", "Original Path"),
        zorder=6,
    )
    smooth_path_x = [p[0] for p in smooth_path]
    smooth_path_y = [p[1] for p in smooth_path]
    ax1.plot(
        smooth_path_x,
        smooth_path_y,
        "-",
        color="darkgreen",
        linewidth=2,
        label=labels.get("平滑路径", "Smoothed Path"),
        zorder=7,
    )
    ax1.plot(
        x_history,
        y_history,
        "-",
        color="crimson",
        linewidth=2.5,
        label=labels.get("车辆轨迹", "Vehicle Trajectory"),
        zorder=8,
    )
    ax1.scatter(
        [path[0][0]], [path[0][1]], color="green", s=100, marker="*", zorder=9
    )
    ax1.scatter(
        [path[-1][0]], [path[-1][1]], color="red", s=100, marker="*", zorder=9
    )
    ax1.legend(loc="upper left", fontsize=10)

    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(
        t_history,
        v_history,
        "-",
        color="blue",
        linewidth=2,
        label=labels.get("实际速度", "Actual Speed"),
    )
    if (
        target_v_history is not None
        and len(target_v_history) == len(t_history)
    ):
        ax2.plot(
            t_history,
            target_v_history,
            "--",
            color="red",
            linewidth=1.5,
            label=labels.get("目标速度", "Target Speed"),
        )
    ax2.fill_between(t_history, 0, v_history, color="skyblue", alpha=0.3)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Speed [m/s]")
    ax2.set_title(
        labels.get("车辆速度", "Vehicle Speed"),
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend()

    ax3 = plt.subplot(gs[1, 1])
    points = np.array([x_history, y_history]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, t_history[-1])
    line_collection = plt.matplotlib.collections.LineCollection(
        segments, cmap="viridis", norm=norm
    )
    line_collection.set_array(np.array(t_history[:-1]))
    line_collection.set_linewidth(3)
    line = ax3.add_collection(line_collection)
    ax3.scatter(
        x_history[0],
        y_history[0],
        color="green",
        s=80,
        marker="o",
        label="Start",
    )
    ax3.scatter(
        x_history[-1],
        y_history[-1],
        color="red",
        s=80,
        marker="o",
        label="Goal",
    )
    ax3.set_xlim(0, env.road_length)
    ax3.set_ylim(0, env.road_width)
    ax3.set_xlabel("X [m]")
    ax3.set_ylabel("Y [m]")
    ax3.set_title(
        labels.get("轨迹时间分布", "Trajectory Time Distribution"),
        fontsize=12,
        fontweight="bold",
    )
    ax3.grid(True, linestyle="--", alpha=0.5)
    cbar = fig.colorbar(line, ax=ax3)
    cbar.set_label("Time [s]")
    ax3.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
