"""
gradient.py —— 策略梯度算法类（REINFORCE，softmax 偏好表）

核心更新公式（来自小车杆prompt.txt）:
    对已执行动作 A_t（提升偏好）:
        H_{t+1}(A_t) ← H_t(A_t) + α (G_t − R̄) (1 − π_t(A_t))
    对所有未执行动作 a ≠ A_t（相对降低偏好）:
        H_{t+1}(a) ← H_t(a) − α (G_t − R̄) π_t(a)
    其中:
        H(s, a)     —— 状态 s 下动作 a 的"偏好值"（越大越倾向选）
        π_t(a)      —— softmax 概率: π(a|s) = e^{H(s,a)} / Σ_a' e^{H(s,a')}
        G_t         —— 从 t 时刻开始的实际回报（未来奖励折扣和）
        R̄           —— 平均回报基线（减去它降低方差，加速收敛）

与 Q-Learning / SARSA 的区别:
    那两个算法学的是"状态-动作的价值表 Q"；
    策略梯度学的是"状态-动作的偏好表 H"，
    偏好经过 softmax 直接变成选动作的概率分布 —— 策略本身。

使用方法（在 fram_work 的训练循环中）:
    每步只记录 (state, action, reward)，不做即时更新；
    一轮结束后调用 update_episode() 统一用回报 G_t 更新整轮的偏好。
"""

import numpy as np


class GradientAgent:
    """基于 softmax 偏好表的 REINFORCE 智能体"""

    def __init__(self, state_shape, n_actions,
                 alpha=0.01, gamma=0.99):
        """
        参数:
            state_shape: 每维箱数组成的元组，如 CartPole 的 (8, 8, 8, 8)
            n_actions:   动作数量（离散动作空间大小）
            alpha:       学习率 α。注意策略梯度通常比 TD 算法用更小的学习率
                         （因为偏好值没有固定上界，更新幅度容易偏大）
            gamma:       折扣因子 γ
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        # 偏好表 H：形状 = (箱1, 箱2, ..., 箱k, 动作数)，初始全 0（= 均匀随机策略）
        self.h_table = np.zeros(state_shape + (n_actions,))

        # 基线 R̄：历史回报的运行平均值，用于减去方差
        self.baseline = 0.0
        self.n_updates = 0

    def policy(self, state):
        """
        计算状态 s 的动作概率分布 π(·|s) = softmax(H(s, ·))。

        参数:
            state: 离散状态索引元组

        返回:
            长度为 n_actions 的概率数组，和为 1
        """
        h = self.h_table[state]
        # 数值稳定技巧：先减去最大值，再取 exp，结果不变但不会溢出
        h = h - np.max(h)
        exp_h = np.exp(h)
        return exp_h / exp_h.sum()

    def choose_action(self, state):
        """
        按概率分布抽样选择动作（不是贪心！概率小的动作也有机会被选）。

        参数:
            state: 离散状态索引元组

        返回:
            动作编号（int）
        """
        return int(np.random.choice(self.n_actions, p=self.policy(state)))

    def update_episode(self, trajectory):
        """
        一轮结束后，用整轮的轨迹更新偏好表。

        参数:
            trajectory: 列表，元素是 (state, action, reward)，
                        按时间顺序记录本轮每一步。
                        即 [(S_0, A_0, R_1), (S_1, A_1, R_2), ...]

        更新流程:
            1. 从后往前累加折扣回报 G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ...
            2. 对每一步，按公式更新 H(S_t, ·)
            3. 更新基线 R̄（运行平均）
        """
        # ---------- 1. 从后往前计算每一步的回报 G_t ----------
        returns = []
        g = 0.0
        for _, _, reward in reversed(trajectory):
            g = reward + self.gamma * g
            returns.append(g)
        returns.reverse()  # 恢复时间顺序：returns[t] = G_t

        # ---------- 2. 逐时间步更新偏好表 ----------
        for (state, action, _), G_t in zip(trajectory, returns):
            pi = self.policy(state)

            # 关键：用【更新前】的旧基线计算差值，再更新基线。
            # 若先更新基线再计算 (G_t − R̄)，当前样本会被自己抵消，
            # 导致第一个 episode 完全学不到东西（手算对照发现的 bug）。
            diff = G_t - self.baseline

            # 更新已执行动作 A_t：表现好（G_t > R̄）就提升偏好
            self.h_table[state][action] += self.alpha * diff * (1 - pi[action])

            # 更新其他动作：相对降低偏好（总和守恒，保证概率分布正常）
            for a in range(self.n_actions):
                if a != action:
                    self.h_table[state][a] -= self.alpha * diff * pi[a]

            # 更新基线（运行平均，越往后越稳定）——放在偏好更新之后
            self.n_updates += 1
            self.baseline += (G_t - self.baseline) / self.n_updates

            # 附注：也可以把 G_t 换成单步奖励 R，就退化成 prompt.txt 里的
            #       H ← H + α(R_t − R̄)(1 − π) 形式；用 G_t 是完整版 REINFORCE。
