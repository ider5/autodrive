"""自动驾驶仿真系统命令行入口。"""

import argparse
import random

import numpy as np

from autodrive import i18n
from autodrive.simulation.session import run_session


def main():
    """解析选项并启动一次交互式仿真。"""
    parser = argparse.ArgumentParser(description="自动驾驶仿真系统")
    parser.add_argument("--seed", type=int, default=None, help="随机数种子")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    print("自动驾驶系统启动...")

    print("\n请选择路径规划算法:")
    print("1. RRT (快速随机树)")
    print("2. A* (A星算法)")
    print("3. RRT* (优化随机树 - 具有渐近最优性)")

    while True:
        try:
            planning_choice = int(input("请输入数字 (1、2 或 3): "))
            if planning_choice in [1, 2, 3]:
                break
            print("请输入有效的数字 1、2 或 3")
        except ValueError:
            print("请输入有效的数字 1、2 或 3")

    print("\n请选择控制器类型:")
    print("1. Pure Pursuit (纯跟踪控制器)")
    print("2. MPC (模型预测控制器)")
    print("3. Stanley (Stanley路径跟踪控制器) [推荐]")

    while True:
        try:
            controller_choice = int(input("请输入数字 (1、2 或 3): "))
            if controller_choice in [1, 2, 3]:
                break
            print("请输入有效的数字 1、2 或 3")
        except ValueError:
            print("请输入有效的数字 1、2 或 3")

    i18n.set_chinese_font()
    if i18n.labels is None:
        i18n.use_english_labels()

    run_session(planning_choice, controller_choice)


if __name__ == "__main__":
    main()
