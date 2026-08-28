"""
utils.py —— 公共工具类

提供强化学习实验中通用的工具函数。
当前核心功能：把"连续状态空间"离散化成"离散索引"，
因为 Q-Learning / SARSA / 表格型 REINFORCE 都是查表算法，
只能处理离散状态（表下标），必须先对连续观测分箱。

设计原则（来自小车杆prompt.txt）：
  简单、可读性优先，只实现必要功能。
"""

import numpy as np


def build_bins(bounds, bins_per_dim):
    """
    为每个状态维度生成分箱边界。

    参数:
        bounds:        每维状态的范围，格式 [(min, max), (min, max), ...]
        bins_per_dim:  每维要分成的箱数，格式 [n1, n2, ...]

    返回:
        bins: 边界数组列表。np.digitize(v, b) 用 b 作边界把 v 映射到 0~n-1 的箱编号。

    原理说明:
        想把一维状态分成 n 个箱，只需要 n-1 个内部边界。
        例如范围 [0, 4] 分成 4 箱，边界取 [1, 2, 3]：
          v < 1   → 箱 0
          1 ≤ v < 2 → 箱 1
          2 ≤ v < 3 → 箱 2
          v ≥ 3   → 箱 3
        即使 v 超出 [min, max]，np.digitize 也不会报错，
        而是自动归入最边上的箱，天然防越界。
    """
    bins = []
    for (low, high), n in zip(bounds, bins_per_dim):
        # 等间距取 n-1 个内部边界
        edges = np.linspace(low, high, n + 1)[1:-1]
        bins.append(edges)
    return bins


def discretize(obs, bins):
    """
    把连续观测向量转换成离散状态索引（元组）。

    参数:
        obs:  环境的连续观测，如 [cart_pos, cart_vel, pole_angle, ...]
        bins: build_bins() 生成的边界列表

    返回:
        元组形式的离散索引，可直接用作 Q 表 / 偏好表的下标，
        如 (3, 5, 2, 7)。

    说明:
        np.digitize 对每个维度独立分箱，
        len(bins[i]) = n-1 个边界 → 该维输出 0~n-1 共 n 个箱编号。
    """
    return tuple(int(np.digitize(v, b)) for v, b in zip(obs, bins))
