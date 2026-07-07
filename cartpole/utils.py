"""
=====================================================================
   工具模块 — 环境创建 & 状态离散化
   将被 cartpole_rl.py / QLearningAgent.py / SARSA.py 共享使用
=====================================================================
"""

import numpy as np
import gymnasium as gym


# ── CartPole-v1 四个状态维度的取值范围 ──────────────────
# 每个元素为 (最小值, 最大值)，用于离散化前将状态值裁剪到合理范围
STATE_RANGES = [
    (-2.4, 2.4),     # 维度0：小车在轨道上的位置（超过 ±2.4 游戏结束）
    (-3.0, 3.0),     # 维度1：小车移动速度（无硬性限制，这里取经验范围）
    (-0.209, 0.209), # 维度2：杆子与垂直线的夹角（弧度），超过 ±12° 游戏结束
    (-2.0, 2.0),     # 维度3：杆子顶端角速度（无硬性限制，经验范围）
]


def make_env(render_human: bool = False):
    """
    创建一个新的 CartPole-v1 环境实例。
    每次训练循环调用一次，避免跨回合的状态残留。
    参数 render_human: 设为 True 则弹出图形窗口展示小车运动
    """
    return gym.make('CartPole-v1', render_mode='human' if render_human else None)


def _build_bins(n_bins: tuple) -> list:
    """
    根据 n_bins 中每个维度的箱子数，生成等距分箱边界。
    例如 n_bins[0]=6，则在 [-2.4, 2.4] 间均匀插入 5 个边界点，
    去掉首尾后得到 5 个内部边界：[-1.6, -0.8, 0.0, 0.8, 1.6]。
    np.digitize 用这 5 个边界将连续值映射到 [0, 1, 2, 3, 4, 5] 共 6 个箱。
    返回：列表，每个元素是一个一维 NumPy 数组（该维度的分箱边界）
    """
    bins = []
    for i, n in enumerate(n_bins):
        lo, hi = STATE_RANGES[i]          # 该维度的 (最小值, 最大值)
        # 在 [lo, hi] 间均匀生成 n+1 个点，去掉最小点和最大点
        # 剩下 n-1 个内部边界，np.digitize 用到这些边界
        bins.append(np.linspace(lo, hi, n + 1)[1:-1])
    return bins


def discretize(state: np.ndarray, bins: list) -> tuple:
    """
    将 4 维连续状态向量映射为离散元组索引，供 Q 表 / H 表查询。
    步骤：
      1. 将每个维度的值裁剪到 STATE_RANGES 范围内
      2. 用 np.digitize 将裁剪后的值映射到箱子编号
      3. 返回四元组如 (3, 2, 7, 4)
    参数：
      state: Gymnasium 返回的长度为 4 的连续状态数组
      bins: 由 _build_bins() 生成的边界列表
    返回：
      四元组，每个元素在 [0, n_bins[i]-1] 范围内
    """
    idx = []
    for i in range(len(state)):
        lo, hi = STATE_RANGES[i]          # 该维度的合法范围
        # 裁剪：将超出范围的值拉回到边界（避免极值导致索引越界）
        clipped = np.clip(state[i], lo, hi)
        # digitize：返回 clipped 落在 bins[i] 中哪个区间
        d = np.digitize(clipped, bins[i])
        # 防止 d 等于 len(bins[i])（即落在最后一个边界右边）
        d = min(d, len(bins[i]))
        idx.append(d)
    return tuple(idx)
