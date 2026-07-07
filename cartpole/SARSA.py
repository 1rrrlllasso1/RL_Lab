
# ============================================================
#  第4部分：SARSA 智能体
#  算法：Q(S,A) ← Q(S,A) + α[R + γ·Q(S',A') - Q(S,A)]
#  特点：在策略（on-policy），使用实际执行的下一个动作的 Q 值
# ============================================================
import numpy as np
from Config import Config
from utils import _build_bins, discretize

class SarsaAgent:
    """SARSA 智能体：用 Q 表 + ε-贪婪策略 + 在策略 TD 更新"""

    def __init__(self, cfg: Config):
        # 与 QLearningAgent 结构相同，因为都用 Q 表
        self.cfg = cfg
        self.bins = _build_bins(cfg.n_bins)
        self.q_table = np.zeros(cfg.n_bins + (2,))
        self.lr = cfg.learning_rate
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """与 Q-Learning 完全相同的 ε-贪婪策略"""
        s = discretize(state, self.bins)
        if training and np.random.random() < self.epsilon:
            return np.random.randint(2)
        return int(np.argmax(self.q_table[s]))

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA 更新公式（在策略）：
          Q(S,A) ← Q(S,A) + α [ R + γ·Q(S',A') - Q(S,A) ]
        关键区别：用的是实际执行的 next_action 的 Q 值，而非最大 Q 值。
        因此调用方需要预先选好 next_action 再传入。
        done=True 时同样 Q(S',A')=0。
        """
        s = discretize(state, self.bins)            # 当前状态索引
        ns = discretize(next_state, self.bins)      # 下一状态索引

        # TD 目标：reward + γ * Q(next_state, next_action)
        td_target = reward + (0 if done else self.gamma * self.q_table[ns + (next_action,)])

        # TD 误差并更新
        td_error = td_target - self.q_table[s + (action,)]
        self.q_table[s + (action,)] += self.lr * td_error

    def decay_epsilon(self):
        """与 Q-Learning 相同的探索率衰减"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
