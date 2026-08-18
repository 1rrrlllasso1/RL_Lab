"""
PPO —— Proximal Policy Optimization 算法类

在 Policy Gradient (REINFORCE, gradient.py) 的基础上加入两项核心改进：

  1. 重要性采样比率（Importance Sampling Ratio）
        r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
     允许用旧策略 π_old 收集的数据多次更新新策略 π_θ，
     大幅提高样本利用率（REINFORCE 的每条数据只用一次）。

  2. Clipped surrogate objective（裁剪代理目标）
        L^CLIP(θ) = E[ min( r_t·Â_t, clip(r_t, 1-ε, 1+ε)·Â_t ) ]
     当 r_t 偏离 1 过远（策略更新过大）时，把目标裁剪到
     [1-ε, 1+ε]·Â_t 范围内，限制单次更新的步长，
     防止策略一步更新过大导致性能崩塌。

  结构：Actor-Critic
    - Actor（策略网络 π_θ）：输入观测，输出每个动作的 logits，
      softmax 后按概率采样动作。
    - Critic（价值网络 V_φ）：输入观测，输出状态价值 V(s)，
      用于计算优势估计。

  优势估计：GAE（Generalized Advantage Estimation）
        δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
        Â_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
    λ 在 0 到 1 之间折中方差与偏差：
    λ=0 退化为 TD(0)（低方差高偏差），λ=1 退化为蒙特卡洛（高方差低偏差）。

  更新流程（每收集满 rollout_steps 步触发一次）：
    1. 用当前 Critic 对整段滚动缓冲区计算 GAE 优势 Â 与回报 G
    2. 对同一批数据做 epochs 轮小批量梯度下降（多次复用数据）
    3. 清空缓冲区，重新收集下一批

  手动可改参数（在本类 __init__ 方法中修改）：
    alpha         学习率 (默认 3e-4，Adam 优化器，神经网络通常很小)
    gamma         折扣因子，衡量未来奖励的重要性 (默认 0.99)
    lam           GAE 的 λ 参数 (默认 0.95)
    clip_ratio    PPO 裁剪阈值 ε (默认 0.2)
    epochs        每次更新遍历滚动缓冲区的轮数 K (默认 10)
    rollout_steps 每次更新前收集的步数（滚动缓冲区大小，默认 1024）
    batch_size    每次小批量梯度下降的样本数 (默认 64)
    hidden_dim    隐藏层神经元数量 (默认 128)
    entropy_coef  熵正则系数，鼓励探索 (默认 0.01)
    epsilon       保留参数，与 SARSA/DQN 接口兼容
                  (PPO 使用 softmax 随机策略探索，不使用 ε-greedy)

  使用方法：
    agent = PPO()
    a = agent.choose_action(obs)               # softmax 策略采样
    agent.update(obs, a, r, next_obs, done)    # 存储一步，缓冲区满后更新

  与 Policy Gradient (gradient.py) 的关键区别：
    - 使用神经网络（Actor + Critic），而非偏好 H 表
    - 用重要性采样比率 + 裁剪目标，支持多轮次复用旧数据
    - 用 GAE 估计优势，比 REINFORCE 的整轮蒙特卡洛回报方差更低

  与 DQN (dqn.py) 的关键区别：
    - 属于策略梯度家族（on-policy），而非值函数拟合（off-policy）
    - 不用目标网络 / 经验回放，而是滚动缓冲区收集一段轨迹后整体更新
    - 动作由随机策略采样来探索，DQN 用 ε-greedy
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPO:
    def __init__(self, n_actions: int = 2, alpha: float = 3e-4,
                 gamma: float = 0.99, lam: float = 0.7,
                 clip_ratio: float = 0.1, epochs: int = 10,
                 rollout_steps: int = 2048, batch_size: int = 256,
                 hidden_dim: int = 128, entropy_coef: float = 0.01,
                 epsilon: float = 0.1):
        """
        初始化 PPO 算法

        参数:
            n_actions:    动作空间大小（CartPole-v1: 0=左, 1=右，共 2 个）
            alpha:        学习率 —— 【手动可改】(默认 3e-4，神经网络通常很小)
            gamma:        折扣因子 —— 【手动可改】(默认 0.99)
            lam:          GAE 的 λ 参数 —— 【手动可改】(默认 0.95)
            clip_ratio:   PPO 裁剪阈值 ε —— 【手动可改】(默认 0.2)
            epochs:       每次更新遍历数据的轮数 K —— 【手动可改】
            rollout_steps: 滚动缓冲区大小（步数）—— 【手动可改】
            batch_size:   小批量大小 —— 【手动可改】
            hidden_dim:   隐藏层神经元数量 —— 【手动可改】
            entropy_coef: 熵正则系数 —— 【手动可改】
            epsilon:      保留参数，与 SARSA/DQN 接口兼容 (不使用 ε-greedy)
        """
        self.n_actions = n_actions
        self.alpha = alpha              # 【手动可改】学习率
        self.gamma = gamma              # 【手动可改】折扣因子
        self.lam = lam                  # 【手动可改】GAE λ
        self.clip_ratio = clip_ratio    # 【手动可改】裁剪阈值
        self.epochs = epochs            # 【手动可改】每批数据复用轮数
        self.rollout_steps = rollout_steps  # 【手动可改】滚动缓冲区大小
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim    # 【手动可改】隐藏层神经元数量
        self.entropy_coef = entropy_coef  # 【手动可改】熵正则系数
        self.epsilon = epsilon          # 【手动可改】软性保留，PPO 不使用 ε-greedy

        # ========== 观测维度 ==========
        # CartPole-v1: 4 维连续空间 [x, x_dot, θ, θ_dot]
        self._obs_dim = 4

        # ========== Actor / Critic 网络 ==========
        # Actor 输出动作概率分布（用于选动作），Critic 输出状态价值 V(s)
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # ========== 优化器（Adam） ==========
        # 用一个优化器同时优化 Actor 与 Critic 两组参数
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=alpha)

        # ========== 损失函数（价值网络 MSE） ==========
        self.loss_fn = nn.MSELoss()

        # ========== 滚动缓冲区（Rollout Buffer） ==========
        # 按步收集 (obs, action, log_prob, reward, done, next_obs)，
        # 收集满 rollout_steps 步后统一做一次 PPO 更新。
        self._buffer_obs = []
        self._buffer_actions = []
        self._buffer_logprobs = []
        self._buffer_rewards = []
        self._buffer_dones = []
        self._buffer_next_obs = []

        # ========== 上一次采样动作的 log 概率 ==========
        # choose_action 时计算并暂存，随后的 update() 写入缓冲区。
        # 重要性采样比率需要"旧策略"下的概率，必须在更新前记录。
        self._last_logprob = None

    # ---------- 内部方法 ----------

    def _build_actor(self) -> nn.Module:
        """
        构建策略网络（Actor）。

        输入观测，输出每个动作的 logits（softmax 后得到 π(·|s)）。
        网络结构（与 DQN 相同的隐藏层配置）：
            输入层(4) → 全连接(hidden_dim) → ReLU →
            全连接(hidden_dim) → ReLU →
            全连接(n_actions) → 输出 logits

        返回:
            nn.Sequential 前馈神经网络
        """
        return nn.Sequential(
            nn.Linear(self._obs_dim, self.hidden_dim),   # 输入层 → 隐藏层 1
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), # 隐藏层 1 → 隐藏层 2
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_actions)   # 隐藏层 2 → 输出层
        )

    def _build_critic(self) -> nn.Module:
        """
        构建价值网络（Critic）。

        结构与 Actor 相同，只是输出为标量 V(s)。
            输入层(4) → 全连接(hidden_dim) → ReLU →
            全连接(hidden_dim) → ReLU →
            全连接(1) → 输出标量价值

        返回:
            nn.Sequential 前馈神经网络
        """
        return nn.Sequential(
            nn.Linear(self._obs_dim, self.hidden_dim),   # 输入层 → 隐藏层 1
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), # 隐藏层 1 → 隐藏层 2
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)                # 隐藏层 2 → 输出层
        )

    # ---------- 公开方法 ----------

    def choose_action(self, obs) -> int:
        """
        从 softmax 策略 π_θ(·|s) 采样动作，并暂存其 log 概率。

        与 PolicyGradient 相同，PPO 不使用 ε-greedy：
        softmax 概率分布本身决定了探索程度，
        随着训练深入策略会自然趋于确定性。

        参数:
            obs: 原始观测值（float32 ndarray，由 obs_to_state 返回）

        返回:
            action: 0（左推）或 1（右推）
        """
        # 将观测转换为 PyTorch 张量，增加 batch 维度
        obs_tensor = torch.FloatTensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():                               # 推理模式，不计算梯度
            logits = self.actor(obs_tensor)                 # shape: (1, n_actions)
            dist = Categorical(logits=logits)
            action = dist.sample()                          # 按概率采样
            # 暂存 log 概率：PPO 更新时计算重要性采样比率 r_t 需要它
            self._last_logprob = dist.log_prob(action).item()
        return int(action.item())

    def update(self, obs, action: int, reward: float, next_obs, done: bool):
        """
        存储一步经验；缓冲区收集满 rollout_steps 步后执行 PPO 更新。

        由于 PPO 需要一段轨迹计算 GAE 优势，且同一批数据要复用
        多轮（epochs 次）训练，更新只能在缓冲区满时进行。
        每步调用该方法只做缓存，缓存满时触发 _learn()。

        参数:
            obs:      当前状态 s（原始观测，由 obs_to_state 返回）
            action:   当前动作 a
            reward:   执行 a 后得到的即时奖励 r
            next_obs: 下一个状态 s'（done 时为 None）
            done:     本轮是否结束
        """
        # ========== 处理 next_obs ==========
        # done=True 时 fram_work 传入 None，用零向量占位；
        # GAE 计算中 (1-done) 会让引导项归零，next_obs 的数值不影响结果。
        if next_obs is None:
            next_obs = np.zeros(self._obs_dim, dtype=np.float32)

        # ========== 存储一步经验 ==========
        self._buffer_obs.append(np.asarray(obs, dtype=np.float32))
        self._buffer_actions.append(int(action))
        self._buffer_logprobs.append(self._last_logprob)
        self._buffer_rewards.append(float(reward))
        self._buffer_dones.append(done)
        self._buffer_next_obs.append(np.asarray(next_obs, dtype=np.float32))

        # ========== 缓冲区满 → 执行 PPO 更新 ==========
        if len(self._buffer_obs) >= self.rollout_steps:
            self._learn()
            self._clear_buffer()

    # ---------- 私有方法 ----------

    def _learn(self):
        """
        对滚动缓冲区执行一次完整的 PPO 更新。

        流程：
            1. 用当前 Critic 计算整段轨迹的价值 V(s)
            2. 计算 GAE 优势 Â 与回报 G（G = Â + V(s)）
            3. 将缓冲区数据打乱，做 epochs 轮小批量梯度下降：
               策略损失（裁剪代理目标） + 0.5·价值损失(MSE) - 熵正则
            4. 优化器更新 Actor 与 Critic 参数

        PPO 裁剪目标：
            r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)   （重要性采样比率）
            L^CLIP = -E[ min(r_t·Â_t, clip(r_t, 1-ε, 1+ε)·Â_t) ]
        """
        # ---- 从缓冲区取数据，转成 numpy 数组 ----
        obs = np.array(self._buffer_obs)                 # (T, 4)
        actions = np.array(self._buffer_actions)         # (T,)
        old_logprobs = np.array(self._buffer_logprobs)   # (T,)
        rewards = np.array(self._buffer_rewards)         # (T,)
        dones = np.array(self._buffer_dones, dtype=np.float32)  # (T,)
        next_obs = np.array(self._buffer_next_obs)       # (T, 4)

        obs_t = torch.FloatTensor(obs)
        next_obs_t = torch.FloatTensor(next_obs)

        # ---- 1. 计算价值与 GAE 优势 ----
        with torch.no_grad():
            values = self.critic(obs_t).squeeze(-1).numpy()        # (T,)
            next_values = self.critic(next_obs_t).squeeze(-1).numpy()

        # TD 误差：δ_t = r_t + γ·V(s')·(1-done) - V(s)
        deltas = rewards + self.gamma * next_values * (1 - dones) - values

        # GAE：Â_t = δ_t + γλ·(1-done)·Â_{t+1}（从后往前递推）
        advantages = np.zeros_like(deltas)
        gae = 0.0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values   # 回报估计 G_t，供价值网络拟合

        # ---- 转 PyTorch 张量 ----
        adv_t = torch.FloatTensor(advantages)
        ret_t = torch.FloatTensor(returns)
        old_logprobs_t = torch.FloatTensor(old_logprobs)
        act_t = torch.LongTensor(actions)

        # 优势归一化（标准技巧，稳定训练）
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ---- 2. 多轮次小批量梯度下降 ----
        n_samples = len(obs)
        for _ in range(self.epochs):
            perm = np.random.permutation(n_samples)   # 打乱顺序
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch_obs = obs_t[idx]
                batch_act = act_t[idx]
                batch_old = old_logprobs_t[idx]
                batch_adv = adv_t[idx]
                batch_ret = ret_t[idx]

                # 新策略 π_θ 下的 log 概率与熵
                dist = Categorical(logits=self.actor(batch_obs))
                new_logprobs = dist.log_prob(batch_act)
                entropy = dist.entropy().mean()

                # 重要性采样比率 r_t(θ) = exp(logπ_θ - logπ_old)
                ratio = torch.exp(new_logprobs - batch_old)

                # ---- 裁剪代理目标（策略损失） ----
                # r_t 在 [1-ε, 1+ε] 内时不受影响；
                # 超出范围（更新过大）时被裁剪，防止策略崩塌。
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio,
                                    1 + self.clip_ratio) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- 价值损失（MSE，拟合回报 G_t） ----
                # critic 输出 shape (batch, 1)，squeeze 成 (batch,) 再与目标对齐
                value_loss = self.loss_fn(self.critic(batch_obs).squeeze(-1), batch_ret)

                # ---- 总损失 = 策略损失 + 0.5·价值损失 - 熵系数·熵 ----
                # 熵正则鼓励探索，防止策略过早收敛到局部最优
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()   # 清空上一轮梯度
                loss.backward()              # 反向传播
                self.optimizer.step()        # 更新参数

    def _clear_buffer(self):
        """清空滚动缓冲区，准备收集下一批数据"""
        self._buffer_obs.clear()
        self._buffer_actions.clear()
        self._buffer_logprobs.clear()
        self._buffer_rewards.clear()
        self._buffer_dones.clear()
        self._buffer_next_obs.clear()

    # ---------- 辅助方法 ----------

    def set_epsilon(self, epsilon: float):
        """
        与 SARSA/DQN 接口兼容的方法。

        PPO 不使用 ε-greedy 探索（由 softmax 随机策略本身负责），
        因此调用此方法不产生实际效果。保留以保持接口一致性。

        参数:
            epsilon: 新的探索率（被忽略）
        """
        pass  # PPO 无需 ε-greedy 探索

    def obs_to_state(self, observation):
        """
        工具方法：PPO 使用原始连续观测，不需要离散化。

        与 DQN 一致，直接返回 float32 的 ndarray，
        保持与现有接口一致（SARSA 等返回离散索引，PPO 返回连续数组）。
        返回的结果可以直接传给 choose_action() 和 update()。

        参数:
            observation: CartPole 返回的原始观测 [x, x_dot, θ, θ_dot]

        返回:
            np.ndarray shape (4,) float32，与输入相同的观测值
        """
        return np.array(observation, dtype=np.float32)
