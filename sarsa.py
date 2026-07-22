"""
SARSA —— 算法类

基于 ε-greedy 策略选择动作，并用 SARSA 更新规则更新 Q 表。
SARSA 是在线策略（on-policy）算法：更新时使用的下一个动作 A' 就是下一步
实际要执行的动作。

更新公式（课本标准写法）：
    Q(S,A) ← Q(S,A) + α[R + γ Q(S',A') - Q(S,A)]
    其中 A' 由当前 ε-greedy 策略从 S' 选出

手动可改参数（在本类 __init__ 方法中修改）：
    alpha   学习率，控制每次更新的步长 (默认 0.1)
    gamma   折扣因子，衡量未来奖励的重要性 (默认 0.99)
    epsilon ε-greedy 中的探索率 (默认 0.1)
    n_bins  每个观测维度离散化的分箱数，传 int 统一指定或传 list of 4 分别指定 (默认 10)
    ranges  每个维度的取值范围，list of 4 (lo, hi) 或 None 使用默认值

使用方法：
    agent = SARSA()                           # 创建 SARSA 智能体
    a = agent.choose_action(state_idx)        # ε-greedy 选动作
    agent.update(s, a, r, s_next, a_next, done)  # 更新 Q 表
"""

import numpy as np
from utils import build_bins, discretize


class SARSA:
    def __init__(self, n_actions: int = 2, alpha: float = 0.1,
                 gamma: float = 0.999, epsilon: float = 0.1,
                 n_bins=10, ranges=None):
        """
        初始化 SARSA 算法

        参数:
            n_actions: 动作空间大小（CartPole-v1: 0=左, 1=右，共 2 个）
            alpha:     学习率 —— 【手动可改】
            gamma:     折扣因子 —— 【手动可改】
            epsilon:   探索率  —— 【手动可改】
            n_bins:    每个观测维度离散化时的分箱数。
                       int 则所有维度相同；list of 4 则分别为
                       [pos, vel, angle, ang_vel] 指定不同分箱数。
            ranges:    每个维度的取值范围，list of 4 (lo, hi) 或
                       None（使用默认值）。
        """
        self.n_actions = n_actions
        self.alpha = alpha          # 【手动可改】学习率 (0 < alpha ≤ 1)
        self.gamma = gamma          # 【手动可改】折扣因子 (0 ≤ gamma ≤ 1)
        self.epsilon = epsilon      # 【手动可改】探索率 (0 ≤ epsilon ≤ 1)
        self.n_bins = n_bins        # 【手动可改】分箱数 (内存：仅此一次哦)

        # ========== 展开 n_bins 为各维度列表 ==========
        if isinstance(n_bins, int):
            self._n_bins_list = [n_bins] * 4
        else:
            self._n_bins_list = list(n_bins)

        # ========== Q 表 ==========
        # 形状：(状态总数, 动作数)
        # 状态总数 = Π n_i（各维度分箱数之积）
        total_states = 1
        for nb in self._n_bins_list:
            total_states *= nb
        self.Q = np.zeros((total_states, n_actions))

        # ========== 离散化分箱边界 ==========
        # 由 utils.build_bins 统一构建，支持自定义范围和分箱数
        self.bins = build_bins(self._n_bins_list, ranges)

    # ---------- 公开方法 ----------

    def choose_action(self, state_idx: int) -> int:
        """
        ε-greedy 策略选择动作。

        规则：
          - 以概率 ε 随机探索（rand() < epsilon）
          - 以概率 1-ε 选择当前 Q 值最大的动作（利益最大化）

        参数:
            state_idx: 离散化后的状态索引

        返回:
            action: 0（左推）或 1（右推）
        """
        if np.random.rand() < self.epsilon:
            # === 探索：随机选择一个动作 ===
            action = np.random.randint(self.n_actions)
        else:
            # === 利益最大化：选择 Q 值最大的动作 ===
            # 如果两个动作 Q 值相同，argmax 默认返回第一个，不影响学习
            action = int(np.argmax(self.Q[state_idx]))

        return action

    def update(self, state_idx: int, action: int, reward: float,
               next_state_idx: int, next_action: int, done: bool):
        """
        用 SARSA 更新规则更新 Q 表。

        公式（实现的就是这个公式）：
            Q(S,A) ← Q(S,A) + α[R + γ Q(S',A') - Q(S,A)]

        参数:
            state_idx:     当前状态 S（离散索引）
            action:        当前动作 A
            reward:        执行 A 后得到的即时奖励 R
            next_state_idx:下一个状态 S'（离散索引）
            next_action:   下一个动作 A'（由当前策略在 S' 选出）
            done:          本轮是否结束（如果结束，则 Q(S',A') = 0）
        """
        # 当前 Q 值
        current_q = self.Q[state_idx, action]

        # 计算目标 Q 值（TD target）
        if done:
            # 如果本轮结束，没有未来奖励，目标 = 当前奖励
            target = reward
        else:
            # SARSA 使用 Q(S',A') —— 即下一个动作 A' 的 Q 值
            target = reward + self.gamma * self.Q[next_state_idx, next_action]

        # 更新 Q 表：向目标靠近一步（步长由 alpha 控制）
        self.Q[state_idx, action] += self.alpha * (target - current_q)

    # ---------- 辅助方法 ----------

    def set_epsilon(self, epsilon: float):
        """
        动态调整探索率 ε。

        可在训练过程中调用，实现 ε 衰减（exploration decay）：
        例如每 500 轮减小一次探索率，让智能体逐渐从探索转向利用。

        参数:
            epsilon: 新的探索率，范围 [0, 1]
        """
        self.epsilon = max(0.0, min(1.0, epsilon))

    def obs_to_state(self, observation):
        """
        工具方法：将环境的连续观测转换为离散状态索引。

        参数:
            observation: CartPole 返回的原始观测 [x, x_dot, θ, θ_dot]

        返回:
            state_idx: 离散状态索引，可直接传给 choose_action() 和 update()
        """
        return discretize(observation, self.bins)
