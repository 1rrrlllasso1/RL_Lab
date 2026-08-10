"""
Q_learning.py —— Q-Learning 算法类（off-policy 时序差分）

核心更新公式（来自小车杆prompt.txt）:
    Q(S,A) ← Q(S,A) + α [ R + γ·max_a' Q(S',a') − Q(S,A) ]

特点:
    - 离策略（off-policy）：更新用的"下一步价值"是 Q(S') 的**最大值**，
      不关心实际执行了哪个动作 —— 相当于在想象"如果走最优动作会怎样"。
    - 探索用 ε-greedy：大部分时候选当前最优，小概率随机探索。
    - Q 表维度 = (各维箱数..., 动作数)，靠 utils.discretize 把连续状态变下标。

使用方法（在 fram_work 的训练循环中）:
    action = agent.choose_action(state)            # 选动作
    agent.update(state, action, reward, next_state, done)  # 更新 Q 表
"""

import numpy as np


class QLearningAgent:
    """基于 Q 表的 Q-Learning 智能体"""

    def __init__(self, state_shape, n_actions,
                 alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        参数:
            state_shape:    每维箱数组成的元组，如 CartPole 的 (8, 8, 8, 8)
            n_actions:      动作数量（离散动作空间大小）
            alpha:          学习率 α，Q 值每次吸收多少新信息（0~1）
            gamma:          折扣因子 γ，未来奖励的打折程度（越接近1越看重远期）
            epsilon:        初始探索率（1.0 = 完全随机，适合训练初期）
            epsilon_min:    探索率下限，防止后期完全失去探索能力
            epsilon_decay:  每轮训练后探索率的衰减系数（乘上去）
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q 表：形状 = (箱1, 箱2, ..., 箱k, 动作数)，全部初始化为 0
        self.q_table = np.zeros(state_shape + (n_actions,))

    def choose_action(self, state):
        """
        ε-greedy 策略选择动作。

        参数:
            state: 离散状态索引元组，如 (3, 5, 2, 7)

        返回:
            动作编号（int）
        """
        # 以 ε 概率随机探索
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        # 否则选 Q 值最大的动作（贪心）
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning 更新公式:
            Q(S,A) ← Q(S,A) + α [ R + γ·max_a' Q(S',a') − Q(S,A) ]

        参数:
            state:      当前状态 S（离散索引）
            action:     执行的动作 A
            reward:     执行后得到的奖励 R
            next_state: 下一状态 S'
            done:       是否终止（本轮结束）。若结束，下一状态没有价值可言，
                        目标值只剩 R 本身。
        """
        # 下一步的最佳价值：若本轮已结束则为 0，否则取 Q(S') 的最大值
        if done:
            best_next_value = 0.0
        else:
            best_next_value = np.max(self.q_table[next_state])

        # TD 目标 = 即时奖励 + 折扣后的未来最佳价值
        target = reward + self.gamma * best_next_value

        # Q 值向目标方向迈一小步（α 控制步长）
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def decay_epsilon(self):
        """每轮训练结束后调用，让探索率逐渐下降（越来越信任学到的策略）"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
