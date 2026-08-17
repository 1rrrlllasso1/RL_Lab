"""
utils —— 工具函数模块

提供其他模块可能用到的公共工具方法。
当前包含：
  - build_bins() : 为连续状态离散化构建分箱边界
  - discretize() : 将连续观测值转换为离散状态索引
  - ReplayBuffer : 经验回放缓冲区（供 DQN 等神经网络算法使用）

设计原则：函数无副作用，输入输出清晰，可被多个算法类复用。
"""

import random
import numpy as np
import torch


def build_bins(n_bins=10, ranges=None):
    """
    为 CartPole 的 4 维连续观测空间构建分箱边界。

    每个观测维度的默认取值范围：
      - cart_pos(水平位置):        [-4.8,   4.8 ]
      - cart_vel(水平速度):        [-5.0,   5.0 ]   (实际无界，此处经验截断)
      - pole_angle(倾斜角度):      [-0.418, 0.418]  (≈ ±24°)
      - pole_angular_vel(角速度):  [-5.0,   5.0 ]   (实际无界，此处经验截断)

    参数:
        n_bins: int 或 list of 4 ints
                每个维度的分箱数。传 int 则所有维度相同，传 list
                则可分别为 [pos, vel, angle, ang_vel] 指定不同分箱数。
        ranges: list of 4 (lo, hi) tuples 或 None
                每个维度的取值范围。None 则使用上述默认值。

    返回:
        bins: list，包含 4 个 numpy 数组，每个数组长度为（对应维度的分箱数 - 1）
    """
    if isinstance(n_bins, int):
        n_bins = [n_bins] * 4

    if ranges is None:
        ranges = [(-4.8, 4.8), (-5.0, 5.0), (-0.418, 0.418), (-5.0, 5.0)]

    bins = []
    for nb, (lo, hi) in zip(n_bins, ranges):
        bins.append(np.linspace(lo, hi, nb - 1))
    return bins


def discretize(observation, bins):
    """
    将连续观测值转换为离散状态索引（展平为单一整数）。

    原理：
      对每个维度 i：
        用 np.digitize 找到观测值落在哪个箱（bin_idx ∈ [0, n_bins_i]）
        将各维度编码为混合基（mixed-radix）整数：
          state = (...(b0 * n_0 + b1) * n_1 + b2) * n_2 + b3

      其中 n_i 是第 i 个维度的分箱数。
      支持每个维度分箱数不同的情况。

    参数:
        observation: 来自环境的观测，形如 [x, x_dot, θ, θ_dot]
        bins:        build_bins() 的返回值

    返回:
        state_idx: 整数，范围 [0, Π n_i)，即各维度分箱数之积
    """
    state_idx = 0

    for i, obs_val in enumerate(observation):
        # 当前维度的分箱数 = 边界数 + 1
        n_bins = len(bins[i]) + 1
        # np.digitize 返回 obs_val 在 bins[i] 中属于哪个区间
        bin_idx = np.digitize(obs_val, bins[i])
        # 混合基编码累加
        state_idx = state_idx * n_bins + bin_idx

    return state_idx


# ============================================================
#  经验回放缓冲区 (Experience Replay Buffer)
#  供 DQN 等基于神经网络的强化学习算法复用
# ============================================================

class ReplayBuffer:
    """
    经验回放缓冲区。

    存储智能体的经验 (s, a, r, s', done) 元组。
    训练时从中随机采样 mini-batch，打破经验之间的时间相关性，
    提高神经网络训练的稳定性。

    当存储量达到 capacity 时，新的经验会覆盖最旧的经验（循环队列）。

    使用方法：
        buffer = ReplayBuffer(capacity=50000)
        buffer.push(obs, action, reward, next_obs, done)
        batch = buffer.sample(batch_size=64)  # 返回 5 个张量
    """

    def __init__(self, capacity: int = 50000):
        """
        初始化缓冲区。

        参数:
            capacity: 缓冲区最大容量，超过后覆盖最旧经验
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0  # 当前写入位置

    def push(self, obs, action: int, reward: float, next_obs, done: bool):
        """
        存储一条经验 (s, a, r, s', done)。

        参数:
            obs:     当前状态 s（numpy 数组）
            action:  当前动作 a
            reward:  即时奖励 r
            next_obs:下一个状态 s'（numpy 数组）
            done:    是否结束
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, action, reward, next_obs, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """
        从缓冲区中随机采样一个 batch。

        参数:
            batch_size: 采样数量

        返回:
            (obs_b, action_b, reward_b, next_obs_b, done_b)
            均为 PyTorch 张量，分别对应：
              - obs_b, next_obs_b:  shape (batch_size, obs_dim)
              - action_b:           shape (batch_size,) 的 LongTensor
              - reward_b, done_b:   shape (batch_size,) 的 FloatTensor
        """
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)

        return (
            torch.FloatTensor(np.array(obs)),           # (batch, obs_dim)
            torch.LongTensor(np.array(action)),          # (batch,)
            torch.FloatTensor(np.array(reward)),         # (batch,)
            torch.FloatTensor(np.array(next_obs)),       # (batch, obs_dim)
            torch.FloatTensor(np.array(done, dtype=float)),  # (batch,)
        )

    def __len__(self) -> int:
        """返回当前缓冲区中的经验数量"""
        return len(self.buffer)
