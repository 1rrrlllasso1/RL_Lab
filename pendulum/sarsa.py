"""
sarsa.py —— SARSA 算法类（on-policy 时序差分）

核心更新公式（来自小车杆prompt.txt）:
    Q(S,A) ← Q(S,A) + α [ R + γ·Q(S',A') − Q(S,A) ]

特点:
    - 在策略（on-policy）：更新用的"下一步价值"是**实际执行的动作 A'** 的 Q 值，
      而不是最大值。A' 由当前 ε-greedy 策略选出，可能带有探索。
    - 与 Q-Learning 的唯一区别就在这：
        * Q-Learning: max_a' Q(S',a')   —— 乐观，想象最优
        * SARSA:      Q(S',A')          —— 务实，用实际动作
      因此在 ε 较大时 SARSA 更保守，Q-Learning 更激进。

使用方法（在 fram_work 的训练循环中）:
    action = agent.choose_action(state)              # 选当前动作
    next_action = agent.choose_action(next_state)    # 还要选下一个动作（SARSA 特有）
    agent.update(state, action, reward, next_state, next_action, done)
"""

import numpy as np


class SarsaAgent:
    """基于 Q 表的 SARSA 智能体"""

    def __init__(self, state_shape, n_actions,
                 alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        参数:
            state_shape:    每维箱数组成的元组，如 CartPole 的 (8, 8, 8, 8)
            n_actions:      动作数量（离散动作空间大小）
            alpha:          学习率 α（0~1）
            gamma:          折扣因子 γ
            epsilon:        初始探索率
            epsilon_min:    探索率下限
            epsilon_decay:  每轮探索率衰减系数
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q 表：形状 = (箱1, 箱2, ..., 箱k, 动作数)
        self.q_table = np.zeros(state_shape + (n_actions,))

    def choose_action(self, state):
        """ε-greedy 策略选择动作（与 Q-Learning 完全相同）"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA 更新公式:
            Q(S,A) ← Q(S,A) + α [ R + γ·Q(S',A') − Q(S,A) ]

        参数:
            state:       当前状态 S
            action:      当前动作 A
            reward:      奖励 R
            next_state:  下一状态 S'
            next_action: 下一状态实际选择的动作 A'（SARSA 的核心，on-policy）
            done:        是否终止。若结束，A' 不存在，价值为 0
        """
        # 下一步的价值：若本轮结束则为 0，否则取"实际动作 A'"的 Q 值
        if done:
            next_value = 0.0
        else:
            next_value = self.q_table[next_state][next_action]

        # TD 目标 = 即时奖励 + 折扣后的下一步价值
        target = reward + self.gamma * next_value

        # Q 值向目标方向更新
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def decay_epsilon(self):
        """每轮训练结束后调用，让探索率逐渐下降"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
