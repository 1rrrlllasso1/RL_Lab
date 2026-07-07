# ============================================================
#  第3部分：Q-Learning 智能体
#  算法：Q(S,A) ← Q(S,A) + α[R + γ·maxₐQ(S',a) - Q(S,A)]
#  特点：离策略（off-policy），使用下一状态的最大 Q 值更新当前 Q
# ============================================================
import numpy as np
from Config import Config
from utils import _build_bins, discretize

class QLearningAgent:
    """Q-Learning 智能体：用 Q 表 + ε-贪婪策略 + 离策略 TD 更新"""

    def __init__(self, cfg: Config):
        # 保存配置引用
        self.cfg = cfg
        # 构建分箱边界（状态离散化用）
        self.bins = _build_bins(cfg.n_bins)
        # Q 表：形状为 (n_bins[0], n_bins[1], n_bins[2], n_bins[3], 2) 的全零数组
        # 前 4 维是离散化后的状态索引，最后一维是动作（0=左推，1=右推）
        self.q_table = np.zeros(cfg.n_bins + (2,))
        # 超参复制到实例变量，方便使用
        self.lr = cfg.learning_rate     # 学习率 α
        self.gamma = cfg.gamma          # 折扣因子 γ
        self.epsilon = cfg.epsilon      # 当前探索率（会逐回合衰减）
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        ε-贪婪策略选择动作：
          - 以概率 ε 随机选（探索）
          - 以概率 1-ε 选 Q 值最大的动作（利用）
        参数 training=False 时强制利用（评估模式）
        """
        s = discretize(state, self.bins)          # 连续状态 → 离散索引
        if training and np.random.random() < self.epsilon:
            return np.random.randint(2)            # 探索：随机返回 0 或 1
        return int(np.argmax(self.q_table[s]))     # 利用：取当前状态 Q 值最大的动作

    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning 更新公式（离策略）：
          Q(S,A) ← Q(S,A) + α [ R + γ·maxₐ Q(S',a) - Q(S,A) ]
        done=True 时目标值仅为 R（因为 S' 是终止状态，Q(S')=0）
        """
        s = discretize(state, self.bins)           # 当前状态索引
        ns = discretize(next_state, self.bins)     # 下一状态索引

        # 计算 TD 目标：reward + γ * maxₐ Q(next_state, a)
        # 如果游戏结束（done=True），则目标值就是 reward（没有下一状态了）
        td_target = reward + (0 if done else self.gamma * np.max(self.q_table[ns]))

        # TD 误差 = 目标值 - 当前 Q 值
        td_error = td_target - self.q_table[s + (action,)]

        # Q 表更新：向目标值靠近一步（步长为 α）
        self.q_table[s + (action,)] += self.lr * td_error

    def decay_epsilon(self):
        """每回合结束后衰减探索率：ε ← max(ε_min, ε * ε_decay)"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
