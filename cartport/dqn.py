"""
DQN —— 深度 Q 网络算法类

使用神经网络来拟合 Q 函数 Q(s,a; θ)，而非传统 Q 表。
更新规则采用 Q-learning（离线策略）：
    target = r + γ * max_a' Q(s', a'; θ_target)
    loss = MSE(Q(s, a; θ_pred) - target)

核心组件：
  - 预测网络 Q(s,a; θ_pred)：   当前用于动作选择和计算 loss
  - 目标网络 Q(s,a; θ_target)： 用于计算 TD target，参数定期从 θ_pred 复制
  - 经验回放缓冲区 ReplayBuffer：存储 (s,a,r,s',done) 元组，随机采样打破相关性

与 Q-learning 的关键区别：
  - 不使用分箱离散化，直接接收原始连续观测
  - 通过经验回放和 mini-batch 更新网络参数
  - 使用目标网络稳定训练

使用方法：
    agent = DQN()                                       # 创建 DQN 智能体
    a = agent.choose_action(obs_array)                  # ε-greedy 选动作
    agent.update(obs, a, r, next_obs, done)             # 存储经验并训练网络
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import ReplayBuffer


class DQN:
    def __init__(self, n_actions: int = 2, alpha: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 hidden_dim: int = 128, buffer_size: int = 50000,
                 batch_size: int = 64, target_update_freq: int = 100):
        """
        初始化 DQN 算法

        参数:
            n_actions:          动作空间大小（CartPole-v1: 0=左, 1=右，共 2 个）
            alpha:              学习率 —— 【手动可改】(默认 0.001，神经网络通常更小的学习率)
            gamma:              折扣因子 —— 【手动可改】
            epsilon:            探索率  —— 【手动可改】
            hidden_dim:         隐藏层神经元数量 —— 【手动可改】
            buffer_size:        经验回放缓冲区容量 —— 【手动可改】
            batch_size:         每次训练采样的 batch 大小 —— 【手动可改】
            target_update_freq: 目标网络更新频率（步数）—— 【手动可改】
        """
        self.n_actions = n_actions
        self.alpha = alpha          # 【手动可改】学习率 (神经网络通常小于 0.01)
        self.gamma = gamma          # 【手动可改】折扣因子 (0 ≤ gamma ≤ 1)
        self.epsilon = epsilon      # 【手动可改】探索率 (0 ≤ epsilon ≤ 1)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # ========== 观测维度 ==========
        # CartPole-v1: 4 维连续空间 [x, x_dot, θ, θ_dot]
        self._obs_dim = 4

        # ========== 神经网络 ==========
        # 预测网络：用于动作选择和训练时的 Q 值计算
        self.q_network = self._build_network(hidden_dim)
        # 目标网络：用于计算 TD target，参数定期从预测网络复制
        self.target_network = self._build_network(hidden_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # ========== 优化器（Adam） ==========
        # Adam 优化器，自适应学习率，适合训练神经网络
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)

        # ========== 损失函数 ==========
        self.loss_fn = nn.MSELoss()

        # ========== 经验回放缓冲区 ==========
        # 由 utils.ReplayBuffer 提供，按 capacity 大小循环存储
        self.replay_buffer = ReplayBuffer(buffer_size)

        # ========== 步数计数器 ==========
        # 用于控制目标网络的定期同步
        self._step_count = 0

    # ---------- 内部方法 ----------

    def _build_network(self, hidden_dim: int) -> nn.Module:
        """
        构建 Q 网络。

        网络结构（CartPole 环境足够简单，两层隐藏层即可）：
            输入层(4) → 全连接(hidden_dim) → ReLU →
            全连接(hidden_dim) → ReLU →
            全连接(n_actions) → 输出

        参数:
            hidden_dim: 隐藏层神经元数量

        返回:
            nn.Sequential 前馈神经网络
        """
        return nn.Sequential(
            nn.Linear(self._obs_dim, hidden_dim),   # 输入层 → 隐藏层 1
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),       # 隐藏层 1 → 隐藏层 2
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_actions)    # 隐藏层 2 → 输出层
        )

    # ---------- 公开方法 ----------

    def choose_action(self, obs: np.ndarray) -> int:
        """
        ε-greedy 策略选择动作。

        规则：
          - 以概率 ε 随机探索
          - 以概率 1-ε 选择当前 Q 值最大的动作

        参数:
            obs: 原始观测值（由 obs_to_state 返回的 float32 ndarray）

        返回:
            action: 0（左推）或 1（右推）
        """
        if np.random.rand() < self.epsilon:
            # === 探索：随机选择一个动作 ===
            return np.random.randint(self.n_actions)

        # === 利益最大化：通过 Q 网络选择动作 ===
        # 将观测转换为 PyTorch 张量，增加 batch 维度
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)   # shape: (1, 4)
        with torch.no_grad():                               # 推理模式，不计算梯度
            q_values = self.q_network(obs_tensor)           # shape: (1, 2)
        return int(torch.argmax(q_values).item())

    def update(self, obs, action: int, reward: float, next_obs, done: bool):
        """
        存储一步经验，并从回放缓冲区采样 batch 训练网络。

        更新规则（与 Q-learning 相同的思想，但基于神经网络）：
            target = r + γ * max_a' Q(s', a'; θ_target)  (if not done)
            loss = MSE(Q(s, a; θ_pred) - target)

        参数:
            obs:      当前状态 s（原始观测，由 obs_to_state 返回）
            action:   当前动作 a
            reward:   执行 a 后得到的即时奖励 r
            next_obs: 下一个状态 s'（由 obs_to_state 返回，done 时为 None）
            done:     本轮是否结束
        """
        # ========== 处理 next_obs ==========
        # 当 done=True 时，fram_work 传入的 next_obs 为 None
        # 此时用零向量填充，但 done 标志会在目标计算中使 γ 项归零，
        # 因此 next_obs 的具体数值不影响结果。
        if done or next_obs is None:
            next_obs = np.zeros(self._obs_dim, dtype=np.float32)

        # ========== 存储经验 ==========
        self.replay_buffer.push(obs, action, reward, next_obs, done)

        self._step_count += 1

        # ========== 训练网络 ==========
        # 经验足够时才训练，否则前几步只收集经验
        if len(self.replay_buffer) >= self.batch_size:
            batch = self.replay_buffer.sample(self.batch_size)
            self._learn(batch)

    # ---------- 私有方法 ----------

    def _learn(self, batch):
        """
        对采样的 batch 执行一次梯度下降更新。

        算法流程：
            1. 通过预测网络计算当前状态-动作的 Q 值 Q(s, a; θ_pred)
            2. 通过目标网络计算目标值 r + γ * max_{a'} Q(s', a'; θ_target)
            3. 计算 MSE loss
            4. 反向传播，更新预测网络参数
            5. 每隔 target_update_freq 步，同步目标网络

        参数:
            batch: ReplayBuffer.sample() 返回的 5 个张量
        """
        obs_b, action_b, reward_b, next_obs_b, done_b = batch

        # ---- 1. 计算当前 Q 值 Q(s, a; θ_pred) ----
        # 网络输出 shape: (batch, n_actions)，选取 a 对应的 Q 值
        q_values = self.q_network(obs_b)                           # (batch, n_actions)
        q_value = q_values.gather(1, action_b.unsqueeze(1)).squeeze(1)  # (batch,)

        # ---- 2. 计算目标 Q 值 ----
        with torch.no_grad():
            # 用目标网络计算下一个状态的 Q 值并取最大值
            next_q_values = self.target_network(next_obs_b)        # (batch, n_actions)
            max_next_q, _ = next_q_values.max(dim=1)                # (batch,)
            # done 时目标仅为当前奖励，无未来回报
            target = reward_b + self.gamma * max_next_q * (1 - done_b)

        # ---- 3. 计算 loss 并更新网络 ----
        loss = self.loss_fn(q_value, target)

        self.optimizer.zero_grad()  # 清空上一轮梯度
        loss.backward()              # 反向传播
        self.optimizer.step()        # 更新参数

        # ---- 4. 定期更新目标网络 ----
        # 软更新：直接将预测网络参数复制给目标网络
        if self._step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    # ---------- 辅助方法 ----------

    def set_epsilon(self, epsilon: float):
        """
        动态调整探索率 ε。

        参数:
            epsilon: 新的探索率，范围 [0, 1]
        """
        self.epsilon = max(0.0, min(1.0, epsilon))

    def obs_to_state(self, observation):
        """
        工具方法：DQN 使用原始连续观测值，不需要离散化。

        将 CartPole 返回的 observation 转换为 float32 的 ndarray，
        保持与现有接口一致（SARSA 等返回离散索引，DQN 返回连续数组）。
        返回的结果可以直接传给 choose_action() 和 update()。

        参数:
            observation: CartPole 返回的原始观测 [x, x_dot, θ, θ_dot]

        返回:
            np.ndarray shape (4,) float32，与输入相同的观测值
        """
        return np.array(observation, dtype=np.float32)
