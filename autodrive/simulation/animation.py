"""Animated simulation result rendering."""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from autodrive import i18n


def create_animation(
    env,
    path,
    x_history,
    y_history,
    yaw_history,
    v_history,
    safety_distance=1.5,
    *,
    dt=0.1,
    show=True,
    animate=True,
    filename="vehicle_animation.gif",
):
    """Create and save the legacy vehicle animation."""
    labels = i18n.labels or i18n.use_english_labels()
    fig, ax = plt.subplots(figsize=(14, 7))
    env.plot_environment(ax)
    ax.text(
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
    ax.plot(
        path_x,
        path_y,
        "-",
        color="darkgreen",
        linewidth=2,
        label="Path",
        zorder=5,
    )

    car_length = env.vehicle_length
    car_width = env.vehicle_width
    car = Rectangle(
        (0, 0),
        car_length,
        car_width,
        fc="red",
        ec="black",
        alpha=0.8,
        zorder=10,
    )
    ax.add_patch(car)

    def update_car_position(rect, center_x, center_y, yaw):
        dx = -car_length / 2
        dy = -car_width / 2
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        corner_x = center_x + dx * cos_yaw - dy * sin_yaw
        corner_y = center_y + dx * sin_yaw + dy * cos_yaw
        rect.set_xy((corner_x, corner_y))
        rect.set_angle(np.rad2deg(yaw))
        if hasattr(ax, "center_point"):
            ax.center_point.remove()
        ax.center_point = ax.plot(
            center_x, center_y, "ko", markersize=4
        )[0]

    trajectory = plt.matplotlib.collections.LineCollection(
        [], cmap="inferno", linewidths=2.5, zorder=7
    )
    ax.add_collection(trajectory)
    info_panel = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            alpha=0.8,
            boxstyle="round,pad=0.5",
        ),
        zorder=20,
    )
    title = labels.get("自动驾驶模拟", "Autonomous Driving Simulation")
    ax.set_title(title, fontsize=16, fontweight="bold")

    progress_bar_bg = Rectangle(
        (0, 0), 0, 0, fc="lightgray", ec="black", zorder=9
    )
    progress_bar = Rectangle(
        (0, 0), 0, 0, fc="limegreen", ec=None, zorder=9
    )
    ax.add_patch(progress_bar_bg)
    ax.add_patch(progress_bar)

    def init():
        update_car_position(
            car, x_history[0], y_history[0], yaw_history[0]
        )
        info_panel.set_text("")
        trajectory.set_segments([])
        progress_bar_bg.set_xy((10, env.road_width + 1))
        progress_bar_bg.set_width(env.road_length - 20)
        progress_bar_bg.set_height(0.5)
        progress_bar.set_xy((10, env.road_width + 1))
        progress_bar.set_width(0)
        progress_bar.set_height(0.5)
        return (
            car,
            info_panel,
            trajectory,
            progress_bar_bg,
            progress_bar,
        )

    def animate_frame(index):
        x = x_history[index]
        y = y_history[index]
        yaw = yaw_history[index]
        speed = v_history[index]
        time = index * dt
        update_car_position(car, x, y, yaw)

        if index > 0:
            points = (
                np.array(
                    [x_history[: index + 1], y_history[: index + 1]]
                )
                .T.reshape(-1, 1, 2)
            )
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            trajectory.set_segments(segments)
            trajectory.set_array(np.array(v_history[:index]))
            plt.Normalize(0, max(v_history))

        time_label = labels.get("时间", "Time")
        speed_label = labels.get("速度", "Speed")
        distance_label = labels.get("行驶距离", "Distance")
        if index > 0:
            distance = sum(
                np.hypot(
                    x_history[j + 1] - x_history[j],
                    y_history[j + 1] - y_history[j],
                )
                for j in range(index)
            )
        else:
            distance = 0
        info_panel.set_text(
            f"{time_label}: {time:.1f}s\n"
            f"{speed_label}: {speed:.2f}m/s\n"
            f"{distance_label}: {distance:.1f}m"
        )
        total_time = len(x_history) * dt
        progress = time / total_time
        progress_bar.set_width((env.road_length - 20) * progress)
        return (
            car,
            info_panel,
            trajectory,
            progress_bar_bg,
            progress_bar,
        )

    ani = animation.FuncAnimation(
        fig,
        animate_frame,
        frames=len(x_history),
        init_func=init,
        blit=False,
        interval=50,
    )
    if animate:
        try:
            ani.save(filename, writer="pillow", fps=20, dpi=100)
            print(f"动画已保存为 '{filename}'")
        except Exception as error:
            print(f"保存动画时出错: {error}")
            print("继续显示静态图片...")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return ani
