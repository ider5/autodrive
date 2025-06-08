"""
字体支持模块

本模块提供中文字体设置和多语言标签支持，
确保matplotlib正确显示中文字符。
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import platform
import os


def set_chinese_font():
    """
    设置matplotlib的中文字体
    
    自动检测操作系统并设置合适的中文字体，
    解决matplotlib显示中文时的乱码问题。
    """
    system = platform.system()
    if system == 'Windows':
        # Windows系统
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun']
        for font in font_list:
            try:
                mpl.rcParams['font.family'] = font
                mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                break
            except:
                continue
    elif system == 'Linux':
        # Linux系统
        mpl.rcParams['font.family'] = 'WenQuanYi Micro Hei'
        mpl.rcParams['axes.unicode_minus'] = False
    elif system == 'Darwin':
        # macOS系统
        mpl.rcParams['font.family'] = 'Arial Unicode MS'
        mpl.rcParams['axes.unicode_minus'] = False
    
    # 如果上述字体都不可用，使用英文标签替代
    if mpl.rcParams['font.family'] == 'sans-serif':
        use_english_labels()

def use_english_labels():
    """使用英文标签替代中文"""
    global labels
    labels = {
        '道路环境': 'Road Environment',
        '起点': 'Start',
        '终点': 'End',
        '原始路径': 'Original Path',
        '平滑路径': 'Smoothed Path',
        '车辆轨迹': 'Vehicle Trajectory',
        '路径规划与跟踪': 'Path Planning & Tracking',
        '车辆速度': 'Vehicle Speed',
        'X位置随时间变化': 'X Position vs Time',
        'Y位置随时间变化': 'Y Position vs Time',
        '时间': 'Time',
        '速度': 'Speed'
    }
    return labels

# 全局标签字典
labels = None