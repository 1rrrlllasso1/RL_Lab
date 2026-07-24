"""
gradient —— Policy Gradient (REINFORCE) 算法类

基于 softmax 策略选择动作，并用 REINFORCE 更新规则更新偏好 H 表。
策略梯度方法是基于策略（policy-based）的算法：直接学习策略函数，
而非通过值函数间接得到策略。

更新公式（策略梯度定理 / REINFORCE）：

    ∇ln π(A_t|S_t, θ) —— 对于 softmax 策略，其梯度为：
        ∇ln π(A_t|S_t) = 1 - π_t(A_t)    对已执行动作 A_t
        ∇ln π(a|S_t)   = -π_t(a)          对所有 a ≠ A_t

    REINFORCE 更新规则（G_t 为从 t 开始的实际折扣回报）：
        θ_{t+1} ← θ_t + α * G_t * ∇ln π(A_t|S_t, θ_t)

    写作偏好 H 的更新形式（带基线 R̄ 以降低方差）：
        H_{t+1}(A_t) ← H_t(A_t) + α * (G_t - R̄) * (1 - π_t(A_t))
        H_{t+1}(a)   ← H_t(a)   - α * (G_t - R̄) * π_t(a)      对所有 a ≠ A_t

    向量化实现等价于：
        H[s] ← H[s] + α * (G_t - R̄) * (one_hot(a) - π(·|s))

手动可改参数（在本类 __init__ 方法中修改）：
    alpha   学习率，控制偏好更新的步长 (默认 0.01，比值函数方法更小)
    gamma   折扣因子，衡量未来奖励的重要性 (默认 0.99)
    epsilon 保留参数，与 SARSA 接口兼容 (策略梯度使用 softmax 探索，不使用 ε-greedy)
    n_bins  每个观测维度离散化的分箱数，传 int 统一指定或传 list of 4 分别指定 (默认 10)
    ranges  每个维度的取值范围，list of 4 (lo, hi) 或 None 使用默认值

使用方法：
    agent = PolicyGradient()                      # 创建策略梯度智能体
    a = agent.choose_action(state_idx)            # softmax 策略采样动作
    agent.update(s, a, r, s_next, done)           # 存储一步，结束后更新

与 SARSA / Q-learning 的关键区别：
    - 使用偏好 H 表替代 Q 表（H 初始化为 0）
    - 动作选择由 softmax 策略 π(a|s) ∝ exp(H[s,a]) 完成，而非 ε-greedy
    - 训练为回合制：每步 update() 存储 (s, a, r)，
      当 done=True 时一次性计算完整折扣回报 G_t 并更新整条轨迹
    - 包含基线 R̄（指数移动平均）以降低策略梯度的方差
"""

import numpy as np
from utils import build_bins, discretize


class PolicyGradient:
    def __init__(self, n_actions: int = 2, alpha: float = 0.01,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 n_bins=10, ranges=None):
        """
        初始化 Policy Gradient (REINFORCE) 算法

        参数:
            n_actions: 动作空间大小（CartPole-v1: 0=左, 1=右，共 2 个）
            alpha:     学习率 —— 【手动可改】(默认 0.01，策略梯度通常更小)
            gamma:     折扣因子 —— 【手动可改】
            epsilon:   保留参数，与 SARSA 接口兼容 —— 【手动可改】
            n_bins:    每个观测维度离散化时的分箱数。
                       int 则所有维度相同；list of 4 则分别为
                       [pos, vel, angle, ang_vel] 指定不同分箱数。
            ranges:    每个维度的取值范围，list of 4 (lo, hi) 或
                       None（使用默认值）。
        """
        self.n_actions = n_actions
        self.alpha = alpha          # 【手动可改】学习率 (策略梯度建议比值函数方法小)
        self.gamma = gamma          # 【手动可改】折扣因子 (0 ≤ gamma ≤ 1)
        self.epsilon = epsilon      # 【手动可改】软性保留，softmax 策略不使用 ε-greedy
        self.n_bins = n_bins        # 【手动可改】分箱数

        # ========== 展开 n_bins 为各维度列表 ==========
        if isinstance(n_bins, int):
            self._n_bins_list = [n_bins] * 4
        else:
            self._n_bins_list = list(n_bins)

        # ========== H 表（偏好） ==========
        # 形状：(状态总数, 动作数)
        # 与 Q 表不同，H 表的数值表示对每个动作的"偏好"，
        # 不直接表示动作价值。H 值越大，该动作被选中的概率越高。
        # 初始化为 0（即 uniform 策略）。
        total_states = 1
        for nb in self._n_bins_list:
            total_states *= nb
        self.H = np.zeros((total_states, n_actions))

        # ========== 离散化分箱边界 ==========
        # 由 utils.build_bins 统一构建，支持自定义范围和分箱数
        self.bins = build_bins(self._n_bins_list, ranges)

        # ========== REINFORCE 轨迹缓存 ==========
        # 每轮收集所有 (state, action, reward) 步，
        # 在 done=True 时计算折扣回报并统一更新偏好。
        self._trajectory = []

        # ========== 基线 R̄ ==========
        # 指数移动平均，用于降低策略梯度的方差。
        # 对偏好 H 的更新使用 (G_t - baseline) 而非裸 G_t。
        self._baseline = 0.0

    # ---------- 内部方法 ----------

    def _policy(self, state_idx: int) -> np.ndarray:
        """
        计算给定状态下 softmax 策略的概率分布 π(·|s)。

        公式：
            π(a|s) = exp(H[s,a]) / Σ_b exp(H[s,b])

        参数:
            state_idx: 离散状态索引

        返回:
            probs: shape (n_actions,) 的概率向量
        """
        prefs = self.H[state_idx]
        # 数值稳定性：减去最大值，防止 exp 溢出
        prefs = prefs - np.max(prefs)
        exp_prefs = np.exp(prefs)
        return exp_prefs / np.sum(exp_prefs)

    # ---------- 公开方法 ----------

    def choose_action(self, state_idx: int) -> int:
        """
        从 softmax 策略 π(·|s) 采样动作。

        与 SARSA/Q-learning 的 ε-greedy 不同：
          - 策略梯度方法不使用 ε-greedy，而是根据 softmax 概率分布采样。
          - 越好的动作被选中的概率越高，差的动作也有一定概率被探索。
          - 随着学习进行，偏好值逐渐分化，策略会自然趋于确定性。

        参数:
            state_idx: 离散化后的状态索引

        返回:
            action: 0（左推）或 1（右推）
        """
        probs = self._policy(state_idx)
        # 按 softmax 概率分布采样
        action = int(np.random.choice(self.n_actions, p=probs))
        return action

    def update(self, state_idx: int, action: int, reward: float,
               next_state_idx=None, done: bool = False):
        """
        存储一步经验，并在回合结束时执行 REINFORCE 更新。

        由于 REINFORCE 需要完整的折扣回报 G_t（需要知道整轮的所有奖励），
        更新只能在回合结束时进行。每步调用该方法只做缓存，
        当 done=True 时触发实际更新。

        公式（实现的就是这个公式）：
            H[s] ← H[s] + α * (G_t - R̄) * ∇ln π(A_t|S_t)
            其中 ∇ln π(A_t|S_t) = one_hot(A_t) - π(·|S_t)

        参数:
            state_idx:     当前状态 S_t（离散索引）
            action:        当前动作 A_t
            reward:        执行 A_t 后得到的即时奖励 R_{t+1}
            next_state_idx:下一个状态 S_{t+1}（策略梯度不使用，与 Q_learning 接口兼容）
            done:          本轮是否结束（当 done=True 时执行完整 REINFORCE 更新）
        """
        # ========== 缓存当前步 ==========
        self._trajectory.append((state_idx, action, reward))

        if not done:
            return

        # ========== 回合结束：执行 REINFORCE 更新 ==========

        # ---- 从后往前计算折扣回报 G_t ----
        # G_t = R_{t+1} + γ * R_{t+2} + γ² * R_{t+3} + ...
        G = 0.0

        # 逆序遍历轨迹，边计算回报边更新偏好
        for s, a, r in reversed(self._trajectory):
            # 计算当前步的折扣回报
            G = r + self.gamma * G

            # ---- 更新基线 R̄（指数移动平均） ----
            # R̄ ← R̄ + 0.1 * (G_t - R̄)
            # 使得基线逐渐逼近平均回报水平
            self._baseline += 0.1 * (G - self._baseline)

            # ---- 计算优势估计 δ_t = G_t - R̄ ----
            delta = G - self._baseline

            # ---- 策略梯度更新 ----
            # 计算 softmax 概率 π(·|s)
            pi = self._policy(s)

            # ∇ln π(A_t|S_t) = one_hot(A_t) - π(·|S_t)
            one_hot = np.zeros(self.n_actions)
            one_hot[a] = 1.0
            grad_log = one_hot - pi  # shape (n_actions,)

            # H[s] ← H[s] + α * δ_t * ∇ln π(A_t|S_t)
            self.H[s] += self.alpha * delta * grad_log
            # 等价于手写的两条规则：
            #   H[s, a] += α * δ_t * (1 - π(a|s))    对已执行动作 A_t
            #   H[s, oa] -= α * δ_t * π(oa|s)        对所有 oa ≠ A_t

        # ---- 清理轨迹，准备下一轮 ----
        self._trajectory.clear()

    # ---------- 辅助方法 ----------

    def set_epsilon(self, epsilon: float):
        """
        与 SARSA/Q-learning 接口兼容的方法。

        策略梯度方法不使用 ε-greedy 探索（由 softmax 策略本身处理），
        因此调用此方法不产生实际效果。保留以保持接口一致性。

        参数:
            epsilon: 新的探索率（被忽略）
        """
        pass  # 策略梯度方法无需 ε-greedy 探索

    def obs_to_state(self, observation):
        """
        工具方法：将环境的连续观测转换为离散状态索引。

        参数:
            observation: CartPole 返回的原始观测 [x, x_dot, θ, θ_dot]

        返回:
            state_idx: 离散状态索引，可直接传给 choose_action() 和 update()
        """
        return discretize(observation, self.bins)
