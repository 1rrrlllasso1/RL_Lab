"""
utils —— 工具函数模块

提供其他模块可能用到的公共工具方法。
当前包含：
  - build_bins() : 为连续状态离散化构建分箱边界
  - discretize() : 将连续观测值转换为离散状态索引

设计原则：函数无副作用，输入输出清晰，可被多个算法类复用。
"""

import numpy as np


def build_bins(n_bins: int = 10):
    """
    为 CartPole 的 4 维连续观测空间构建分箱边界。

    每个观测维度的取值范围：
      - cart_pos(水平位置):        [-4.8,   4.8 ]
      - cart_vel(水平速度):        [-5.0,   5.0 ]   (实际无界，此处经验截断)
      - pole_angle(倾斜角度):           [-0.418, 0.418]  (≈ ±24°)
      - pole_angular_vel(角速度):     [-5.0,   5.0 ]   (实际无界，此处经验截断)

    参数:
        n_bins: 每个维度的分箱数（边界数 = n_bins - 1）

    返回:
        bins: list，包含 4 个 numpy 数组，每个数组长度为 n_bins - 1
    """
    bins = [
        np.linspace(-4.8, 4.8, n_bins - 1),       # 小车位置
        np.linspace(-5.0, 5.0, n_bins - 1),        # 小车速度
        np.linspace(-0.418, 0.418, n_bins - 1),    # 杆子角度
        np.linspace(-5.0, 5.0, n_bins - 1),        # 杆子角速度
    ]
    return bins


def discretize(observation, bins):
    """
    将连续观测值转换为离散状态索引（展平为单一整数）。

    原理：
      1. 对每个维度，用 np.digitize 找到它落在哪个箱中
      2. 将 4 个箱编号合并为一个唯一整数：idx = (((b0 * n) + b1) * n + b2) * n + b3
         这等价于把 (b0,b1,b2,b3) 看作 n 进制数的四个数位

    参数:
        observation: 来自环境的观测，形如 [x, x_dot, θ, θ_dot]
        bins:        build_bins() 的返回值

    返回:
        state_idx: 整数，范围 [0, n_bins^4)
    """
    n_bins = len(bins[0]) + 1  # 每个维度的箱子数
    state_idx = 0

    for i, obs_val in enumerate(observation):
        # np.digitize 返回 obs_val 在 bins[i] 中属于哪个区间（1-based）
        bin_idx = np.digitize(obs_val, bins[i])
        # 将当前维度合并到总索引中（类似 n 进制数的进位）
        state_idx = state_idx * n_bins + bin_idx

    return state_idx
