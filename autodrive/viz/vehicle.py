"""Bicycle model visualization."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from autodrive.i18n import set_chinese_font
from autodrive.vehicle import BicycleModel


def visualize_bicycle_model(
    show=True, output_path="vehicle_model_visualization.png"
):
    """可视化自行车模型。"""
    set_chinese_font()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "自行车运动学模型(Bicycle Kinematic Model)可视化", fontsize=16
    )
    axes = axes.flatten()
    vehicle = BicycleModel()

    ax1 = axes[0]
    ax1.set_title("自行车模型基本结构")
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect("equal")
    ax1.grid(True)
    car_length = vehicle.length
    car_width = vehicle.width
    car = Rectangle(
        (-car_length / 2, -car_width / 2),
        car_length,
        car_width,
        angle=0,
        facecolor="lightblue",
        alpha=0.7,
        edgecolor="blue",
        linewidth=2,
    )
    ax1.add_patch(car)
    wheel_width = 0.2
    wheel_length = 0.5
    rear_wheel = Rectangle(
        (
            -car_length / 2 + wheel_length / 2,
            -car_width / 4 - wheel_width / 2,
        ),
        wheel_length,
        wheel_width,
        angle=0,
        facecolor="black",
        alpha=0.8,
    )
    front_wheel = Rectangle(
        (
            car_length / 2 - wheel_length * 1.5,
            -car_width / 4 - wheel_width / 2,
        ),
        wheel_length,
        wheel_width,
        angle=0,
        facecolor="black",
        alpha=0.8,
    )
    ax1.add_patch(rear_wheel)
    ax1.add_patch(front_wheel)
    ax1.plot(
        [-car_length / 4, car_length / 4], [0, 0], "r-", linewidth=2
    )
    ax1.text(-0.5, 0.2, f"轴距 (L): {vehicle.L}m", fontsize=10)
    ax1.plot([-car_length / 4], [0], "ko", markersize=6)
    ax1.plot([car_length / 4], [0], "ko", markersize=6)
    ax1.text(-car_length / 4 - 0.5, -0.4, "后轮", fontsize=10)
    ax1.text(car_length / 4 - 0.3, -0.4, "前轮", fontsize=10)
    ax1.arrow(
        0,
        0,
        1.5,
        0,
        head_width=0.2,
        head_length=0.3,
        fc="green",
        ec="green",
        linewidth=2,
    )
    ax1.text(1.0, 0.3, "yaw (航向角)", fontsize=10, color="green")

    ax2 = axes[1]
    ax2.set_title("不同转向角下的车辆轨迹")
    ax2.set_xlim(-2, 8)
    ax2.set_ylim(-2, 6)
    ax2.set_aspect("equal")
    ax2.grid(True)
    steering_angles = [0, 5, 10, 15, 20]
    colors = ["blue", "green", "red", "purple", "orange"]
    for index, angle in enumerate(steering_angles):
        delta = np.deg2rad(angle)
        x, y, yaw = 0, 0, 0
        speed = 1.0
        trajectory_x = [x]
        trajectory_y = [y]
        for _ in range(30):
            x += speed * np.cos(yaw) * 0.1
            y += speed * np.sin(yaw) * 0.1
            if angle != 0:
                yaw += speed * np.tan(delta) / vehicle.L * 0.1
            trajectory_x.append(x)
            trajectory_y.append(y)
        ax2.plot(
            trajectory_x,
            trajectory_y,
            color=colors[index],
            linewidth=2,
            label=f"转向角: {angle}°",
        )
    ax2.legend(loc="upper left")

    ax3 = axes[2]
    ax3.set_title("自行车模型状态更新方程")
    ax3.axis("off")
    text = "\n".join(
        (
            "状态更新方程:",
            r"$x_{t+1} = x_t + v_t \cdot \cos(\theta_t) \cdot dt$",
            r"$y_{t+1} = y_t + v_t \cdot \sin(\theta_t) \cdot dt$",
            r"$\theta_{t+1} = \theta_t + \frac{v_t \cdot \tan(\delta_t)}{L} \cdot dt$",
            r"$v_{t+1} = v_t + a_t \cdot dt$",
            "",
            "其中:",
            r"$(x, y)$ - 车辆位置坐标",
            r"$\theta$ - 航向角 (yaw)",
            r"$v$ - 车辆速度",
            r"$\delta$ - 前轮转向角",
            r"$a$ - 加速度",
            r"$L$ - 轴距",
            r"$dt$ - 时间步长",
        )
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax3.text(
        0.05,
        0.95,
        text,
        transform=ax3.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )

    ax4 = axes[3]
    ax4.set_title("车辆物理约束参数")
    ax4.axis("off")
    constraints = "\n".join(
        (
            "车辆约束参数:",
            f"• 轴距 (L): {vehicle.L}m",
            f"• 车身尺寸: {vehicle.length}m × {vehicle.width}m",
            f"• 最大转向角: ±{np.rad2deg(vehicle.max_steer):.1f}°",
            f"• 最大转向角速度: {np.rad2deg(vehicle.max_delta_dot):.1f}°/s",
            f"• 速度范围: {vehicle.min_v} - {vehicle.max_v}m/s",
            f"• 最大加速度: ±{vehicle.max_a}m/s²",
            f"• 最大加加速度(jerk): {vehicle.max_jerk}m/s³",
            f"• 时间步长: {vehicle.dt}s",
            "",
            "高级特性:",
            "• 转向角变化率限制",
            "• 加速度变化率限制",
            "• 高速区动态降低可用加速度",
            "• 航向角规范化",
        )
    )
    ax4.text(
        0.05,
        0.95,
        constraints,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"车辆自行车模型可视化已保存为'{output_path}'")
    return output_path
